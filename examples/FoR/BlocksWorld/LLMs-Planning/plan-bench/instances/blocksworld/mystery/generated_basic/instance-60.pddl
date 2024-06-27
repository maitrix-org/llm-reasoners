

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(planet b)
(craves c a)
(craves d c)
(province b)
(province d)
)
(:goal
(and
(craves a c)
(craves b a))
)
)


