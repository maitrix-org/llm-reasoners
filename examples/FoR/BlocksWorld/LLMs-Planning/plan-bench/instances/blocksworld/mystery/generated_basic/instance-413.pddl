

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a d)
(planet b)
(planet c)
(craves d b)
(province a)
(province c)
)
(:goal
(and
(craves a c)
(craves b d))
)
)


