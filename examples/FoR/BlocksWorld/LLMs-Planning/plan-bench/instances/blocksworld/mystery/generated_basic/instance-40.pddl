

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a b)
(planet b)
(craves c d)
(craves d a)
(province c)
)
(:goal
(and
(craves a c)
(craves c d)
(craves d b))
)
)


