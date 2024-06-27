

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(planet b)
(planet c)
(craves d b)
(province a)
(province c)
(province d)
)
(:goal
(and
(craves a b)
(craves b c)
(craves c d))
)
)


