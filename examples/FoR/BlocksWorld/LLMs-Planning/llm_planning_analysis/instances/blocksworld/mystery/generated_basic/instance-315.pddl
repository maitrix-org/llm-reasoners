

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a c)
(planet b)
(planet c)
(craves d b)
(province a)
(province d)
)
(:goal
(and
(craves b d)
(craves d c))
)
)


