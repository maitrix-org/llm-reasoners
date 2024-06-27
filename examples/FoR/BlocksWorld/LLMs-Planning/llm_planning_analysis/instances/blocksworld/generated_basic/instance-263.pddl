

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a c)
(on b a)
(on c d)
(ontable d)
(clear b)
)
(:goal
(and
(on a d)
(on c a)
(on d b))
)
)


