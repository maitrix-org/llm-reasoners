

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a b)
(planet b)
(planet c)
(craves d a)
(province c)
(province d)
)
(:goal
(and
(craves a b)
(craves c d))
)
)


