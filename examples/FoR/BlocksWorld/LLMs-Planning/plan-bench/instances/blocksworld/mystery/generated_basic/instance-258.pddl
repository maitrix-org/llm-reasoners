

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a c)
(planet b)
(planet c)
(craves d a)
(province b)
(province d)
)
(:goal
(and
(craves c d)
(craves d a))
)
)


