

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a c)
(ontable b)
(on c d)
(ontable d)
(clear a)
(clear b)
)
(:goal
(and
(on b c)
(on c a))
)
)


